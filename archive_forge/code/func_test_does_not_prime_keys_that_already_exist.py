from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_does_not_prime_keys_that_already_exist():

    @Promise.safe
    def do():
        identity_loader, load_calls = id_loader()
        identity_loader.prime('A', 'X')
        a1 = identity_loader.load('A').get()
        b1 = identity_loader.load('B').get()
        assert a1 == 'X'
        assert b1 == 'B'
        identity_loader.prime('A', 'Y')
        identity_loader.prime('B', 'Y')
        a2 = identity_loader.load('A').get()
        b2 = identity_loader.load('B').get()
        assert a2 == 'X'
        assert b2 == 'B'
        assert load_calls == [['B']]
    do().get()