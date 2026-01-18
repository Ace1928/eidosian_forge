import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib
def c_preprocess(csource, definitions={}, rownum=0):
    """
    Rudimentary C-preprocessor for ifdef blocks.

    Args:
    - csource: iterator C source code
    - definitions: a mapping (e.g., set or dict contaning
      which "names" are defined)

    Returns:
    The csource with the conditional ifdef blocks for name
    processed.
    """
    localdefs = definitions.copy()
    block = []
    for row in csource:
        rownum += 1
        m = DEFINE_PAT.match(row)
        if m:
            localdefs[m.group(1)] = m.group(2)
            continue
        m_ifdef = IFDEF_PAT.match(row)
        if m_ifdef:
            name = m_ifdef.group(1)
            subblock, subdefs = _c_preprocess_ifdef(csource, name in localdefs, definitions=localdefs, rownum=0)
            block.extend(subblock)
            localdefs.update(subdefs)
            continue
        for k, v in localdefs.items():
            if isinstance(v, str):
                row = row.replace(k, v)
        block.append(row)
    return (block, rownum)