from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_can_represent_failures_and_successes_simultaneously():

    @Promise.safe
    def do():

        def resolve(keys):
            mapped_keys = [key if key % 2 == 0 else Exception('Odd: {}'.format(key)) for key in keys]
            return Promise.resolve(mapped_keys)
        even_loader, load_calls = id_loader(resolve=resolve)
        promise1 = even_loader.load(1)
        promise2 = even_loader.load(2)
        with raises(Exception) as exc_info:
            promise1.get()
        assert str(exc_info.value) == 'Odd: 1'
        value2 = promise2.get()
        assert value2 == 2
        assert load_calls == [[1, 2]]
    do().get()