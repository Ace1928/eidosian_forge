from __future__ import with_statement
import textwrap
from difflib import ndiff
from io import open
from os import listdir
from os.path import dirname, isdir, join, realpath, relpath, splitext
import pytest
import chardet
@pytest.mark.xfail
@given(st.text(min_size=1), st.sampled_from(['ascii', 'utf-8', 'utf-16', 'utf-32', 'iso-8859-7', 'iso-8859-8', 'windows-1255']), st.randoms())
@settings(max_examples=200)
def test_never_fails_to_detect_if_there_is_a_valid_encoding(txt, enc, rnd):
    try:
        data = txt.encode(enc)
    except UnicodeEncodeError:
        assume(False)
    detected = chardet.detect(data)['encoding']
    if detected is None:
        with pytest.raises(JustALengthIssue):

            @given(st.text(), random=rnd)
            @settings(verbosity=Verbosity.quiet, max_shrinks=0, max_examples=50)
            def string_poisons_following_text(suffix):
                try:
                    extended = (txt + suffix).encode(enc)
                except UnicodeEncodeError:
                    assume(False)
                result = chardet.detect(extended)
                if result and result['encoding'] is not None:
                    raise JustALengthIssue()