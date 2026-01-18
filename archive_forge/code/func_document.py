from typing import Dict
def document(obj, doc):
    """Adds a docstring to typealias by overriding the `__doc__` attribute.

  Note: Overriding `__doc__` is only possible after python 3.7.

  Args:
    obj: Typealias object that needs to be documented.
    doc: Docstring of the typealias. It should follow the standard pystyle
      docstring rules.
  """
    try:
        obj.__doc__ = doc
    except AttributeError:
        _EXTRA_DOCS[id(obj)] = doc