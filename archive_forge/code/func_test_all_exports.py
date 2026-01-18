import pytest
@pytest.mark.parametrize('pkgname', ['zmq', 'zmq.green'])
@pytest.mark.parametrize('attr', ['RCVTIMEO', 'PUSH', 'zmq_version_info', 'SocketOption', 'device', 'Socket', 'Context'])
def test_all_exports(pkgname, attr):
    import zmq
    subpkg = pytest.importorskip(pkgname)
    for name in zmq.__all__:
        assert hasattr(subpkg, name)
    assert attr in subpkg.__all__
    if attr not in ('Socket', 'Context', 'device'):
        assert getattr(subpkg, attr) is getattr(zmq, attr)