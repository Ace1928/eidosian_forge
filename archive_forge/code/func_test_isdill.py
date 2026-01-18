import dill as pickle
from io import BytesIO as StringIO
def test_isdill():
    obj_io = StringIO()
    pickler = pickle.Pickler(obj_io)
    assert pickle._dill.is_dill(pickler) is True
    pickler = pickle._dill.StockPickler(obj_io)
    assert pickle._dill.is_dill(pickler) is False
    try:
        import multiprocess as mp
        pickler = mp.reduction.ForkingPickler(obj_io)
        assert pickle._dill.is_dill(pickler, child=True) is True
        assert pickle._dill.is_dill(pickler, child=False) is False
    except Exception:
        pass