from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def processDoctype(self, token):
    name = token['name']
    publicId = token['publicId']
    systemId = token['systemId']
    correct = token['correct']
    if name != 'html' or publicId is not None or (systemId is not None and systemId != 'about:legacy-compat'):
        self.parser.parseError('unknown-doctype')
    if publicId is None:
        publicId = ''
    self.tree.insertDoctype(token)
    if publicId != '':
        publicId = publicId.translate(asciiUpper2Lower)
    if not correct or token['name'] != 'html' or publicId.startswith(('+//silmaril//dtd html pro v0r11 19970101//', '-//advasoft ltd//dtd html 3.0 aswedit + extensions//', '-//as//dtd html 3.0 aswedit + extensions//', '-//ietf//dtd html 2.0 level 1//', '-//ietf//dtd html 2.0 level 2//', '-//ietf//dtd html 2.0 strict level 1//', '-//ietf//dtd html 2.0 strict level 2//', '-//ietf//dtd html 2.0 strict//', '-//ietf//dtd html 2.0//', '-//ietf//dtd html 2.1e//', '-//ietf//dtd html 3.0//', '-//ietf//dtd html 3.2 final//', '-//ietf//dtd html 3.2//', '-//ietf//dtd html 3//', '-//ietf//dtd html level 0//', '-//ietf//dtd html level 1//', '-//ietf//dtd html level 2//', '-//ietf//dtd html level 3//', '-//ietf//dtd html strict level 0//', '-//ietf//dtd html strict level 1//', '-//ietf//dtd html strict level 2//', '-//ietf//dtd html strict level 3//', '-//ietf//dtd html strict//', '-//ietf//dtd html//', '-//metrius//dtd metrius presentational//', '-//microsoft//dtd internet explorer 2.0 html strict//', '-//microsoft//dtd internet explorer 2.0 html//', '-//microsoft//dtd internet explorer 2.0 tables//', '-//microsoft//dtd internet explorer 3.0 html strict//', '-//microsoft//dtd internet explorer 3.0 html//', '-//microsoft//dtd internet explorer 3.0 tables//', '-//netscape comm. corp.//dtd html//', '-//netscape comm. corp.//dtd strict html//', "-//o'reilly and associates//dtd html 2.0//", "-//o'reilly and associates//dtd html extended 1.0//", "-//o'reilly and associates//dtd html extended relaxed 1.0//", '-//softquad software//dtd hotmetal pro 6.0::19990601::extensions to html 4.0//', '-//softquad//dtd hotmetal pro 4.0::19971010::extensions to html 4.0//', '-//spyglass//dtd html 2.0 extended//', '-//sq//dtd html 2.0 hotmetal + extensions//', '-//sun microsystems corp.//dtd hotjava html//', '-//sun microsystems corp.//dtd hotjava strict html//', '-//w3c//dtd html 3 1995-03-24//', '-//w3c//dtd html 3.2 draft//', '-//w3c//dtd html 3.2 final//', '-//w3c//dtd html 3.2//', '-//w3c//dtd html 3.2s draft//', '-//w3c//dtd html 4.0 frameset//', '-//w3c//dtd html 4.0 transitional//', '-//w3c//dtd html experimental 19960712//', '-//w3c//dtd html experimental 970421//', '-//w3c//dtd w3 html//', '-//w3o//dtd w3 html 3.0//', '-//webtechs//dtd mozilla html 2.0//', '-//webtechs//dtd mozilla html//')) or (publicId in ('-//w3o//dtd w3 html strict 3.0//en//', '-/w3c/dtd html 4.0 transitional/en', 'html')) or (publicId.startswith(('-//w3c//dtd html 4.01 frameset//', '-//w3c//dtd html 4.01 transitional//')) and systemId is None) or (systemId and systemId.lower() == 'http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd'):
        self.parser.compatMode = 'quirks'
    elif publicId.startswith(('-//w3c//dtd xhtml 1.0 frameset//', '-//w3c//dtd xhtml 1.0 transitional//')) or (publicId.startswith(('-//w3c//dtd html 4.01 frameset//', '-//w3c//dtd html 4.01 transitional//')) and systemId is not None):
        self.parser.compatMode = 'limited quirks'
    self.parser.phase = self.parser.phases['beforeHtml']