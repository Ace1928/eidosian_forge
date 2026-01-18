from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_build_a_simple_data_loader():

    def call_fn(keys):
        return Promise.resolve(keys)
    identity_loader = DataLoader(call_fn)
    promise1 = identity_loader.load(1)
    assert isinstance(promise1, Promise)
    value1 = promise1.get()
    assert value1 == 1