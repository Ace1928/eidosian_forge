from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_case_str() -> None:
    value = xr.DataArray(['SOme wOrd Ǆ ß ᾛ ΣΣ ﬃ⁵Å Ç Ⅰ']).astype(np.str_)
    exp_capitalized = xr.DataArray(['Some word ǆ ß ᾓ σς ﬃ⁵å ç ⅰ']).astype(np.str_)
    exp_lowered = xr.DataArray(['some word ǆ ß ᾓ σς ﬃ⁵å ç ⅰ']).astype(np.str_)
    exp_swapped = xr.DataArray(['soME WoRD ǆ SS ᾛ σς FFI⁵å ç ⅰ']).astype(np.str_)
    exp_titled = xr.DataArray(['Some Word ǅ Ss ᾛ Σς Ffi⁵Å Ç Ⅰ']).astype(np.str_)
    exp_uppered = xr.DataArray(['SOME WORD Ǆ SS ἫΙ ΣΣ FFI⁵Å Ç Ⅰ']).astype(np.str_)
    exp_casefolded = xr.DataArray(['some word ǆ ss ἣι σσ ffi⁵å ç ⅰ']).astype(np.str_)
    exp_norm_nfc = xr.DataArray(['SOme wOrd Ǆ ß ᾛ ΣΣ ﬃ⁵Å Ç Ⅰ']).astype(np.str_)
    exp_norm_nfkc = xr.DataArray(['SOme wOrd DŽ ß ᾛ ΣΣ ffi5Å Ç I']).astype(np.str_)
    exp_norm_nfd = xr.DataArray(['SOme wOrd Ǆ ß ᾛ ΣΣ ﬃ⁵Å Ç Ⅰ']).astype(np.str_)
    exp_norm_nfkd = xr.DataArray(['SOme wOrd DŽ ß ᾛ ΣΣ ffi5Å Ç I']).astype(np.str_)
    res_capitalized = value.str.capitalize()
    res_casefolded = value.str.casefold()
    res_lowered = value.str.lower()
    res_swapped = value.str.swapcase()
    res_titled = value.str.title()
    res_uppered = value.str.upper()
    res_norm_nfc = value.str.normalize('NFC')
    res_norm_nfd = value.str.normalize('NFD')
    res_norm_nfkc = value.str.normalize('NFKC')
    res_norm_nfkd = value.str.normalize('NFKD')
    assert res_capitalized.dtype == exp_capitalized.dtype
    assert res_casefolded.dtype == exp_casefolded.dtype
    assert res_lowered.dtype == exp_lowered.dtype
    assert res_swapped.dtype == exp_swapped.dtype
    assert res_titled.dtype == exp_titled.dtype
    assert res_uppered.dtype == exp_uppered.dtype
    assert res_norm_nfc.dtype == exp_norm_nfc.dtype
    assert res_norm_nfd.dtype == exp_norm_nfd.dtype
    assert res_norm_nfkc.dtype == exp_norm_nfkc.dtype
    assert res_norm_nfkd.dtype == exp_norm_nfkd.dtype
    assert_equal(res_capitalized, exp_capitalized)
    assert_equal(res_casefolded, exp_casefolded)
    assert_equal(res_lowered, exp_lowered)
    assert_equal(res_swapped, exp_swapped)
    assert_equal(res_titled, exp_titled)
    assert_equal(res_uppered, exp_uppered)
    assert_equal(res_norm_nfc, exp_norm_nfc)
    assert_equal(res_norm_nfd, exp_norm_nfd)
    assert_equal(res_norm_nfkc, exp_norm_nfkc)
    assert_equal(res_norm_nfkd, exp_norm_nfkd)