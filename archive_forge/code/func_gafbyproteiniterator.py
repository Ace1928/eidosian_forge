import copy
def gafbyproteiniterator(handle):
    """Iterate over records in a gene association file.

    Returns a list of all consecutive records with the same DB_Object_ID
    This function should be called to read a
    gene_association.goa_uniprot file. Reads the first record and
    returns a gaf 2.0 or a gaf 1.0 iterator as needed
    2016-04-09: added GAF 2.1 iterator & fixed bug in iterator assignment
    In the meantime GAF 2.1 uses the GAF 2.0 iterator
    """
    inline = handle.readline()
    if inline.strip() == '!gaf-version: 2.0':
        return _gaf20byproteiniterator(handle)
    elif inline.strip() == '!gaf-version: 1.0':
        return _gaf10byproteiniterator(handle)
    elif inline.strip() == '!gaf-version: 2.1':
        return _gaf20byproteiniterator(handle)
    elif inline.strip() == '!gaf-version: 2.2':
        return _gaf20byproteiniterator(handle)
    else:
        raise ValueError(f'Unknown GAF version {inline}\n')