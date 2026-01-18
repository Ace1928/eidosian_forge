@classmethod
def should_shuffle(cls, datatype: str) -> bool:
    """
        Return whether we should shuffle data based on the datatype.

        :param datatype:
            parlai datatype

        :return should_shuffle:
            given datatype, return whether we should shuffle
        """
    assert datatype is not None, 'datatype must not be none'
    return 'train' in datatype and 'evalmode' not in datatype and ('ordered' not in datatype) and ('stream' not in datatype)