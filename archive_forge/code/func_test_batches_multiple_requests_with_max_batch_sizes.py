from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_batches_multiple_requests_with_max_batch_sizes():

    @Promise.safe
    def do():
        identity_loader, load_calls = id_loader(max_batch_size=2)
        promise1 = identity_loader.load(1)
        promise2 = identity_loader.load(2)
        promise3 = identity_loader.load(3)
        p = Promise.all([promise1, promise2, promise3])
        value1, value2, value3 = p.get()
        assert value1 == 1
        assert value2 == 2
        assert value3 == 3
        assert load_calls == [[1, 2], [3]]
    do().get()