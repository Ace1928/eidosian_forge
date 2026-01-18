from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_batches_multiple_requests():

    @Promise.safe
    def do():
        identity_loader, load_calls = id_loader()
        promise1 = identity_loader.load(1)
        promise2 = identity_loader.load(2)
        p = Promise.all([promise1, promise2])
        value1, value2 = p.get()
        assert value1 == 1
        assert value2 == 2
        assert load_calls == [[1, 2]]
    do().get()