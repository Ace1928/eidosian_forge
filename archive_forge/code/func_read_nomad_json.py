from ase.nomad import read as _read_nomad_json
def read_nomad_json(fd, index):
    from ase.io.formats import string2index
    if isinstance(index, str):
        index = string2index(index)
    d = _read_nomad_json(fd)
    images = list(d.iterimages())
    return images[index]