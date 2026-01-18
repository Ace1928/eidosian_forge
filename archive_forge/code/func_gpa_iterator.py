import copy
def gpa_iterator(handle):
    """Read GPA format files.

    This function should be called to read a
    gene_association.goa_uniprot file. Reads the first record and
    returns a gpa 1.1 or a gpa 1.0 iterator as needed
    """
    inline = handle.readline()
    if inline.strip() == '!gpa-version: 1.1':
        return _gpa11iterator(handle)
    elif inline.strip() == '!gpa-version: 1.0':
        return _gpa10iterator(handle)
    else:
        raise ValueError(f'Unknown GPA version {inline}\n')