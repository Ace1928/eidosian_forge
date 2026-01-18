from typing import List, Optional, Union
def highlight(self, fields: Optional[List[str]]=None, tags: Optional[List[str]]=None) -> None:
    """
        Apply specified markup to matched term(s) within the returned field(s).

        - **fields** If specified then only those mentioned fields are
        highlighted, otherwise all fields are highlighted
        - **tags** A list of two strings to surround the match.
        """
    args = ['HIGHLIGHT']
    fields = self._mk_field_list(fields)
    if fields:
        args += ['FIELDS', str(len(fields))] + fields
    if tags:
        args += ['TAGS'] + list(tags)
    self._highlight_fields = args
    return self