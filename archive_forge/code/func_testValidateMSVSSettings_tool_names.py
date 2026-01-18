import unittest
import gyp.MSVSSettings as MSVSSettings
from io import StringIO
def testValidateMSVSSettings_tool_names(self):
    """Tests that only MSVS tool names are allowed."""
    MSVSSettings.ValidateMSVSSettings({'VCCLCompilerTool': {}, 'VCLinkerTool': {}, 'VCMIDLTool': {}, 'foo': {}, 'VCResourceCompilerTool': {}, 'VCLibrarianTool': {}, 'VCManifestTool': {}, 'ClCompile': {}}, self.stderr)
    self._ExpectedWarnings(['Warning: unrecognized tool foo', 'Warning: unrecognized tool ClCompile'])