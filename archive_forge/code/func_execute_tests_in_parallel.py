import unittest
from _pydev_bundle._pydev_saved_modules import thread
import queue as Queue
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
import os
import threading
import sys
def execute_tests_in_parallel(tests, jobs, split, verbosity, coverage_files, coverage_include):
    """
    @param tests: list(PydevTestSuite)
        A list with the suites to be run

    @param split: str
        Either 'module' or the number of tests that should be run in each batch

    @param coverage_files: list(file)
        A list with the files that should be used for giving coverage information (if empty, coverage information
        should not be gathered).

    @param coverage_include: str
        The pattern that should be included in the coverage.

    @return: bool
        Returns True if the tests were actually executed in parallel. If the tests were not executed because only 1
        should be used (e.g.: 2 jobs were requested for running 1 test), False will be returned and no tests will be
        run.

        It may also return False if in debug mode (in which case, multi-processes are not accepted)
    """
    try:
        from _pydevd_bundle.pydevd_comm import get_global_debugger
        if get_global_debugger() is not None:
            return False
    except:
        pass
    tests_queue = []
    queue_elements = []
    if split == 'module':
        module_to_tests = {}
        for test in tests:
            lst = []
            flatten_test_suite(test, lst)
            for test in lst:
                key = (test.__pydev_pyfile__, test.__pydev_module_name__)
                module_to_tests.setdefault(key, []).append(test)
        for key, tests in module_to_tests.items():
            queue_elements.append(tests)
        if len(queue_elements) < jobs:
            jobs = len(queue_elements)
    elif split == 'tests':
        for test in tests:
            lst = []
            flatten_test_suite(test, lst)
            for test in lst:
                queue_elements.append([test])
        if len(queue_elements) < jobs:
            jobs = len(queue_elements)
    else:
        raise AssertionError('Do not know how to handle: %s' % (split,))
    for test_cases in queue_elements:
        test_queue_elements = []
        for test_case in test_cases:
            try:
                test_name = test_case.__class__.__name__ + '.' + test_case._testMethodName
            except AttributeError:
                test_name = test_case.__class__.__name__ + '.' + test_case._TestCase__testMethodName
            test_queue_elements.append(test_case.__pydev_pyfile__ + '|' + test_name)
        tests_queue.append(test_queue_elements)
    if jobs < 2:
        return False
    sys.stdout.write('Running tests in parallel with: %s jobs.\n' % (jobs,))
    queue = Queue.Queue()
    for item in tests_queue:
        queue.put(item, block=False)
    providers = []
    clients = []
    for i in range(jobs):
        test_cases_provider = CommunicationThread(queue)
        providers.append(test_cases_provider)
        test_cases_provider.start()
        port = test_cases_provider.port
        if coverage_files:
            clients.append(ClientThread(i, port, verbosity, coverage_files.pop(0), coverage_include))
        else:
            clients.append(ClientThread(i, port, verbosity))
    for client in clients:
        client.start()
    client_alive = True
    while client_alive:
        client_alive = False
        for client in clients:
            if not client.finished:
                client_alive = True
                time.sleep(0.2)
                break
    for provider in providers:
        provider.shutdown()
    return True