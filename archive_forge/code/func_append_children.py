from suds import *
from suds.umx import *
from suds.umx.attrlist import AttrList
from suds.sax.text import Text
from suds.sudsobject import Factory, merge
def append_children(self, content):
    """
        Append child nodes into L{Content.data}
        @param content: The current content being unmarshalled.
        @type content: L{Content}
        """
    for child in content.node:
        cont = Content(child)
        cval = self.append(cont)
        key = reserved.get(child.name, child.name)
        if key in content.data:
            v = getattr(content.data, key)
            if isinstance(v, list):
                v.append(cval)
            else:
                setattr(content.data, key, [v, cval])
            continue
        if self.multi_occurrence(cont):
            if cval is None:
                setattr(content.data, key, [])
            else:
                setattr(content.data, key, [cval])
        else:
            setattr(content.data, key, cval)