import unittest
import pulp
from pulp.tests import test_pulp, test_examples, test_gurobipy_env
def pulpTestAll(test_docs=False):
    runner = unittest.TextTestRunner()
    suite_all = get_test_suite(test_docs)
    ret = runner.run(suite_all)
    if not ret.wasSuccessful():
        raise pulp.PulpError('Tests Failed')