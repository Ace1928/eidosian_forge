import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def assemble_my_parts(self):
    """Assemble the `self.parts` dictionary.  Extend in subclasses.
        """
    writers.Writer.assemble_parts(self)
    f = tempfile.NamedTemporaryFile()
    zfile = zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED)
    self.write_zip_str(zfile, 'mimetype', self.MIME_TYPE, compress_type=zipfile.ZIP_STORED)
    content = self.visitor.content_astext()
    self.write_zip_str(zfile, 'content.xml', content)
    s1 = self.create_manifest()
    self.write_zip_str(zfile, 'META-INF/manifest.xml', s1)
    s1 = self.create_meta()
    self.write_zip_str(zfile, 'meta.xml', s1)
    s1 = self.get_stylesheet()
    language_code = None
    region_code = None
    if self.visitor.language_code:
        language_ids = self.visitor.language_code.replace('_', '-')
        language_ids = language_ids.split('-')
        language_code = language_ids[0].lower()
        for subtag in language_ids[1:]:
            if len(subtag) == 2 and subtag.isalpha():
                region_code = subtag.upper()
                break
            elif len(subtag) == 1:
                break
        if region_code is None:
            try:
                rcode = locale.normalize(language_code)
            except NameError:
                rcode = language_code
            rcode = rcode.split('_')
            if len(rcode) > 1:
                rcode = rcode[1].split('.')
                region_code = rcode[0]
            if region_code is None:
                self.document.reporter.warning('invalid language-region.\n  Could not find region with locale.normalize().\n  Please specify both language and region (ll-RR).\n  Examples: es-MX (Spanish, Mexico),\n  en-AU (English, Australia).')
    updated, new_dom_styles, updated_node = self.update_stylesheet(self.visitor.get_dom_stylesheet(), language_code, region_code)
    if updated:
        s1 = etree.tostring(new_dom_styles)
    self.write_zip_str(zfile, 'styles.xml', s1)
    self.store_embedded_files(zfile)
    self.copy_from_stylesheet(zfile)
    zfile.close()
    f.seek(0)
    whole = f.read()
    f.close()
    self.parts['whole'] = whole
    self.parts['encoding'] = self.document.settings.output_encoding
    self.parts['version'] = docutils.__version__