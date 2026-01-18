import pickle
import pytest
import numpy as np
def test__size_to_string(self):
    """ Test e._size_to_string """
    f = _ArrayMemoryError._size_to_string
    Ki = 1024
    assert f(0) == '0 bytes'
    assert f(1) == '1 bytes'
    assert f(1023) == '1023 bytes'
    assert f(Ki) == '1.00 KiB'
    assert f(Ki + 1) == '1.00 KiB'
    assert f(10 * Ki) == '10.0 KiB'
    assert f(int(999.4 * Ki)) == '999. KiB'
    assert f(int(1023.4 * Ki)) == '1023. KiB'
    assert f(int(1023.5 * Ki)) == '1.00 MiB'
    assert f(Ki * Ki) == '1.00 MiB'
    assert f(int(Ki * Ki * Ki * 0.9999)) == '1.00 GiB'
    assert f(Ki * Ki * Ki * Ki * Ki * Ki) == '1.00 EiB'
    assert f(Ki * Ki * Ki * Ki * Ki * Ki * 123456) == '123456. EiB'