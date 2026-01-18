import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def patched_to_html(self, *args, **kwargs):
    """A patched version of the to_html method
     that allows rendering molecule images in data frames.
  """
    frame = None
    if self.__class__.__name__ == 'DataFrameRenderer':
        fmt = self.fmt
    elif self.__class__.__name__ == 'DataFrameFormatter':
        fmt = self
    else:
        raise ValueError(f'patched_to_html: unexpected class {self.__class__.__name__}')
    frame = fmt.frame
    if not check_rdk_attr(frame, RDK_MOLS_AS_IMAGE_ATTR):
        return orig_to_html(self, *args, **kwargs)
    orig_formatters = fmt.formatters
    try:
        formatters = orig_formatters or {}
        if not isinstance(formatters, dict):
            formatters = {col: formatters[i] for i, col in enumerate(self.columns)}
        else:
            formatters = dict(formatters)
        formatters.update(MolFormatter.get_formatters(frame, formatters))
        fmt.formatters = formatters
        res = orig_to_html(self, *args, **kwargs)
        if res is None and (not hasattr(html_formatter_class, 'get_result')) and hasattr(self, 'buf') and hasattr(self.buf, 'getvalue'):
            res = self.buf.getvalue()
        should_inject = res and InteractiveRenderer and InteractiveRenderer.isEnabled()
        if should_inject:
            res = InteractiveRenderer.injectHTMLFooterAfterTable(res)
            if hasattr(self, 'buf') and isinstance(self.buf, StringIO):
                self.buf.seek(0)
                self.buf.write(res)
        return res
    finally:
        fmt.formatters = orig_formatters