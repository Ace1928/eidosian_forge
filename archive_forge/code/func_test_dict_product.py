from tune._utils import _EMPTY_ITER, dict_product, product, safe_iter
def test_dict_product():
    res = list(dict_product({}, safe=True))
    assert [{}] == res
    res = list(dict_product({'a': []}, safe=True))
    assert [{}] == res
    res = list(dict_product({}, safe=False))
    assert [] == res
    res = list(dict_product({'a': []}, safe=False))
    assert [] == res