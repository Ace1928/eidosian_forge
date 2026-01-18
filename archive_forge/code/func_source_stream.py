from ... import errors, urlutils
from ...commands import Command
from ...controldir import ControlDir
from ...option import Option
def source_stream():
    for vf_name, keys in needed:
        vf = getattr(source, vf_name)
        yield (vf_name, vf.get_record_stream(keys, 'unordered', True))