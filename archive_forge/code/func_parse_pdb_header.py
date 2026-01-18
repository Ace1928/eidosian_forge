import re
from Bio import File
def parse_pdb_header(infile):
    """Return the header lines of a pdb file as a dictionary.

    Dictionary keys are: head, deposition_date, release_date, structure_method,
    resolution, structure_reference, journal_reference, author and
    compound.
    """
    header = []
    with File.as_handle(infile) as f:
        for line in f:
            record_type = line[0:6]
            if record_type in ('ATOM  ', 'HETATM', 'MODEL '):
                break
            else:
                header.append(line)
    return _parse_pdb_header_list(header)