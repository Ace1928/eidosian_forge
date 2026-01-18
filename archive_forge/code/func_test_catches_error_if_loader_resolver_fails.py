from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_catches_error_if_loader_resolver_fails():

    @Promise.safe
    def do():

        def do_resolve(x):
            raise Exception('AOH!')
        a_loader, a_load_calls = id_loader(resolve=do_resolve)
        with raises(Exception) as exc_info:
            a_loader.load('A1').get()
        assert str(exc_info.value) == 'AOH!'
    do().get()