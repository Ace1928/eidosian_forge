from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_can_call_a_loader_from_a_loader():

    @Promise.safe
    def do():
        deep_loader, deep_load_calls = id_loader()
        a_loader, a_load_calls = id_loader(resolve=lambda keys: deep_loader.load(tuple(keys)))
        b_loader, b_load_calls = id_loader(resolve=lambda keys: deep_loader.load(tuple(keys)))
        a1, b1, a2, b2 = Promise.all([a_loader.load('A1'), b_loader.load('B1'), a_loader.load('A2'), b_loader.load('B2')]).get()
        assert a1 == 'A1'
        assert b1 == 'B1'
        assert a2 == 'A2'
        assert b2 == 'B2'
        assert a_load_calls == [['A1', 'A2']]
        assert b_load_calls == [['B1', 'B2']]
        assert deep_load_calls == [[('A1', 'A2'), ('B1', 'B2')]]
    do().get()