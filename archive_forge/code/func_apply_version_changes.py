from collections import namedtuple
def apply_version_changes():
    """Apply version changes to Structures in FFmpeg libraries.
       Field data can vary from version to version, however assigning _fields_ automatically assigns memory.
       _fields_ can also not be re-assigned. Use a temporary list that can be manipulated before setting the
       _fields_ of the Structure."""
    for library, data in _version_changes.items():
        for version in data:
            for structure, cf_data in _version_changes[library][version].items():
                if versions[library] == version:
                    if cf_data.removals:
                        for remove_field in cf_data.removals:
                            for field in list(cf_data.fields):
                                if field[0] == remove_field:
                                    cf_data.fields.remove(field)
                    if cf_data.repositions:
                        for reposition in cf_data.repositions:
                            data = None
                            insertId = None
                            for idx, field in enumerate(list(cf_data.fields)):
                                if field[0] == reposition.field:
                                    data = field
                                elif field[0] == reposition.after:
                                    insertId = idx
                            if data and insertId:
                                cf_data.fields.remove(data)
                                cf_data.fields.insert(insertId + 1, data)
                            else:
                                print(f'Warning: {reposition} for {library} was not able to be processed.')
                    structure._fields_ = cf_data.fields