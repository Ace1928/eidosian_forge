from promise import Promise
from promise.dataloader import DataLoader
import threading
def task_1():

    @Promise.safe
    def do():
        promise = thread_name_loader.load(1)
        event_1.set()
        event_2.wait()
        assert_object['is_same_thread_1'] = promise.get() == threading.current_thread().getName()
        event_3.set()
    do().get()