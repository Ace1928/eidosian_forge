def decode_numpy(obj, chain=None):
    """
    Decoder for deserializing numpy data types.
    """
    try:
        if b'nd' in obj:
            if obj[b'nd'] is True:
                if b'kind' in obj and obj[b'kind'] == b'V':
                    descr = [tuple((tostr(t) if type(t) is bytes else t for t in d)) for d in obj[b'type']]
                else:
                    descr = obj[b'type']
                return np.frombuffer(obj[b'data'], dtype=np.dtype(descr)).reshape(obj[b'shape'])
            else:
                descr = obj[b'type']
                return np.frombuffer(obj[b'data'], dtype=np.dtype(descr))[0]
        elif b'complex' in obj:
            return complex(tostr(obj[b'data']))
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)