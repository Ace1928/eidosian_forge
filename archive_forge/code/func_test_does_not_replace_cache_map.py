from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_does_not_replace_cache_map():

    @Promise.safe
    def do():
        identity_loader, _ = id_loader()
        a, b = Promise.all([identity_loader.load('A'), identity_loader.load('B')]).get()
        assert a == 'A'
        assert b == 'B'
        cache_map = identity_loader._promise_cache
        identity_loader.clear_all()
        assert id(identity_loader._promise_cache) == id(cache_map)
    do().get()