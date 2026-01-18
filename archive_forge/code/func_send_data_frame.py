import io
import ntpath
import base64
def send_data_frame(writer, filename, type=None, **kwargs):
    """
    Convert data frame into the format expected by the Download component.
    :param writer: a data frame writer
    :param filename: the name of the file
    :param type: type of the file (optional, passed to Blob in the javascript layer)
    :return: dict of data frame content (base64 encoded) and meta data used by the Download component

    Examples
    --------

    >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 1, 5, 6], 'c': ['x', 'x', 'y', 'y']})
    ...
    >>> send_data_frame(df.to_csv, "mydf.csv")  # download as csv
    >>> send_data_frame(df.to_json, "mydf.json")  # download as json
    >>> send_data_frame(df.to_excel, "mydf.xls", index=False) # download as excel
    >>> send_data_frame(df.to_pickle, "mydf.pkl") # download as pickle

    """
    name = writer.__name__
    if name not in _data_frame_senders.keys():
        raise ValueError('The provided writer ({}) is not supported, try calling send_string or send_bytes directly.'.format(name))
    return _data_frame_senders[name](writer, filename, type, **kwargs)