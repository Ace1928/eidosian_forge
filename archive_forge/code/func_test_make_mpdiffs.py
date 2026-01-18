import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
def test_make_mpdiffs(self):
    from breezy import multiparent
    files = self.get_versionedfiles('source')
    files.add_lines(self.get_simple_key(b'base'), [], [b'line\n'])
    files.add_lines(self.get_simple_key(b'noeol'), self.get_parents([self.get_simple_key(b'base')]), [b'line'])
    files.add_lines(self.get_simple_key(b'noeolsecond'), self.get_parents([self.get_simple_key(b'noeol')]), [b'line\n', b'line'])
    files.add_lines(self.get_simple_key(b'noeolnotshared'), self.get_parents([self.get_simple_key(b'noeolsecond')]), [b'line\n', b'phone'])
    files.add_lines(self.get_simple_key(b'eol'), self.get_parents([self.get_simple_key(b'noeol')]), [b'phone\n'])
    files.add_lines(self.get_simple_key(b'eolline'), self.get_parents([self.get_simple_key(b'noeol')]), [b'line\n'])
    files.add_lines(self.get_simple_key(b'noeolbase'), [], [b'line'])
    files.add_lines(self.get_simple_key(b'eolbeforefirstparent'), self.get_parents([self.get_simple_key(b'noeolbase'), self.get_simple_key(b'noeol')]), [b'line'])
    files.add_lines(self.get_simple_key(b'noeoldup'), self.get_parents([self.get_simple_key(b'noeol')]), [b'line'])
    next_parent = self.get_simple_key(b'base')
    text_name = b'chain1-'
    text = [b'line\n']
    sha1s = {0: b'da6d3141cb4a5e6f464bf6e0518042ddc7bfd079', 1: b'45e21ea146a81ea44a821737acdb4f9791c8abe7', 2: b'e1f11570edf3e2a070052366c582837a4fe4e9fa', 3: b'26b4b8626da827088c514b8f9bbe4ebf181edda1', 4: b'e28a5510be25ba84d31121cff00956f9970ae6f6', 5: b'd63ec0ce22e11dcf65a931b69255d3ac747a318d', 6: b'2c2888d288cb5e1d98009d822fedfe6019c6a4ea', 7: b'95c14da9cafbf828e3e74a6f016d87926ba234ab', 8: b'779e9a0b28f9f832528d4b21e17e168c67697272', 9: b'1f8ff4e5c6ff78ac106fcfe6b1e8cb8740ff9a8f', 10: b'131a2ae712cf51ed62f143e3fbac3d4206c25a05', 11: b'c5a9d6f520d2515e1ec401a8f8a67e6c3c89f199', 12: b'31a2286267f24d8bedaa43355f8ad7129509ea85', 13: b'dc2a7fe80e8ec5cae920973973a8ee28b2da5e0a', 14: b'2c4b1736566b8ca6051e668de68650686a3922f2', 15: b'5912e4ecd9b0c07be4d013e7e2bdcf9323276cde', 16: b'b0d2e18d3559a00580f6b49804c23fea500feab3', 17: b'8e1d43ad72f7562d7cb8f57ee584e20eb1a69fc7', 18: b'5cf64a3459ae28efa60239e44b20312d25b253f3', 19: b'1ebed371807ba5935958ad0884595126e8c4e823', 20: b'2aa62a8b06fb3b3b892a3292a068ade69d5ee0d3', 21: b'01edc447978004f6e4e962b417a4ae1955b6fe5d', 22: b'd8d8dc49c4bf0bab401e0298bb5ad827768618bb', 23: b'c21f62b1c482862983a8ffb2b0c64b3451876e3f', 24: b'c0593fe795e00dff6b3c0fe857a074364d5f04fc', 25: b'dd1a1cf2ba9cc225c3aff729953e6364bf1d1855'}
    for depth in range(26):
        new_version = self.get_simple_key(text_name + b'%d' % depth)
        text = text + [b'line\n']
        files.add_lines(new_version, self.get_parents([next_parent]), text)
        next_parent = new_version
    next_parent = self.get_simple_key(b'base')
    text_name = b'chain2-'
    text = [b'line\n']
    for depth in range(26):
        new_version = self.get_simple_key(text_name + b'%d' % depth)
        text = text + [b'line\n']
        files.add_lines(new_version, self.get_parents([next_parent]), text)
        next_parent = new_version
    target = self.get_versionedfiles('target')
    for key in multiparent.topo_iter_keys(files, files.keys()):
        mpdiff = files.make_mpdiffs([key])[0]
        parents = files.get_parent_map([key])[key] or []
        target.add_mpdiffs([(key, parents, files.get_sha1s([key])[key], mpdiff)])
        self.assertEqualDiff(next(files.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext'), next(target.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext'))