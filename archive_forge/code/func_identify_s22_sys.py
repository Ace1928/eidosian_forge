from ase.atoms import Atoms
def identify_s22_sys(name, dist=None):
    s22_ = False
    s22x5_ = False
    if (name in s22 or name in s26) and dist == None:
        s22_name = name
        s22_ = True
    elif name in s22x5 and dist == None:
        s22_name, dist = get_s22x5_id(name)
        s22x5_ = True
    elif name in s22 and dist != None:
        dist_ = str(dist)
        if dist_ not in ['0.9', '1.0', '1.2', '1.5', '2.0']:
            raise KeyError('Bad s22x5 distance specified: %s' % dist_)
        else:
            s22_name = name
            dist = dist_
            s22x5_ = True
    if s22_ is False and s22x5_ is False:
        raise KeyError('s22 combination %s %s not in database' % (name, str(dist)))
    return (s22_, s22x5_, s22_name, dist)