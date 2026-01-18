from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_allows_priming_the_cache():

    @Promise.safe
    def do():
        identity_loader, load_calls = id_loader()
        identity_loader.prime('A', 'A')
        a, b = Promise.all([identity_loader.load('A'), identity_loader.load('B')]).get()
        assert a == 'A'
        assert b == 'B'
        assert load_calls == [['B']]
    do().get()