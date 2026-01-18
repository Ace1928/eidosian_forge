import pytest
import re
from spacy_loggers.util import matcher_for_regex_patterns
from .util import load_logger_from_config
def test_custom_stats_matcher():
    patterns = ['^[pP]ytorch', 'zeppelin$']
    inputs = ['no match', 'torch', 'pYtorch', 'pytorch', 'Pytorch 1.13', 'led zeppelin']
    outputs = [False, False, False, True, True, True]
    matcher = matcher_for_regex_patterns(patterns)
    assert [matcher(x) for x in inputs] == outputs
    with pytest.raises(ValueError, match="couldn't be compiled"):
        matcher_for_regex_patterns([')'])