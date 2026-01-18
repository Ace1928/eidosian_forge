from promise import Promise
from promise.dataloader import DataLoader
import threading
def test_dataloader_thread_safety():
    """
    Dataloader should only batch `load` calls that happened on the same thread.
    
    Here we assert that `load` calls on thread 2 are not batched on thread 1 as
    thread 1 batches its own `load` calls.
    """

    def load_many(keys):
        thead_name = threading.current_thread().getName()
        return Promise.resolve([thead_name for key in keys])
    thread_name_loader = DataLoader(load_many)
    event_1 = threading.Event()
    event_2 = threading.Event()
    event_3 = threading.Event()
    assert_object = {'is_same_thread_1': True, 'is_same_thread_2': True}

    def task_1():

        @Promise.safe
        def do():
            promise = thread_name_loader.load(1)
            event_1.set()
            event_2.wait()
            assert_object['is_same_thread_1'] = promise.get() == threading.current_thread().getName()
            event_3.set()
        do().get()

    def task_2():

        @Promise.safe
        def do():
            promise = thread_name_loader.load(2)
            event_2.set()
            event_3.wait()
            assert_object['is_same_thread_2'] = promise.get() == threading.current_thread().getName()
        do().get()
    thread_1 = threading.Thread(target=task_1)
    thread_1.start()
    event_1.wait()
    thread_2 = threading.Thread(target=task_2)
    thread_2.start()
    for thread in (thread_1, thread_2):
        thread.join()
    assert assert_object['is_same_thread_1']
    assert assert_object['is_same_thread_2']