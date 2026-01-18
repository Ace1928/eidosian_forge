from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_caches_repeated_requests():

    @Promise.safe
    def do():
        identity_loader, load_calls = id_loader()
        a, b = Promise.all([identity_loader.load('A'), identity_loader.load('B')]).get()
        assert a == 'A'
        assert b == 'B'
        assert load_calls == [['A', 'B']]
        a2, c = Promise.all([identity_loader.load('A'), identity_loader.load('C')]).get()
        assert a2 == 'A'
        assert c == 'C'
        assert load_calls == [['A', 'B'], ['C']]
        a3, b2, c2 = Promise.all([identity_loader.load('A'), identity_loader.load('B'), identity_loader.load('C')]).get()
        assert a3 == 'A'
        assert b2 == 'B'
        assert c2 == 'C'
        assert load_calls == [['A', 'B'], ['C']]
    do().get()