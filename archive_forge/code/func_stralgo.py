import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def stralgo(self, algo: Literal['LCS'], value1: KeyT, value2: KeyT, specific_argument: Union[Literal['strings'], Literal['keys']]='strings', len: bool=False, idx: bool=False, minmatchlen: Union[int, None]=None, withmatchlen: bool=False, **kwargs) -> ResponseT:
    """
        Implements complex algorithms that operate on strings.
        Right now the only algorithm implemented is the LCS algorithm
        (longest common substring). However new algorithms could be
        implemented in the future.

        ``algo`` Right now must be LCS
        ``value1`` and ``value2`` Can be two strings or two keys
        ``specific_argument`` Specifying if the arguments to the algorithm
        will be keys or strings. strings is the default.
        ``len`` Returns just the len of the match.
        ``idx`` Returns the match positions in each string.
        ``minmatchlen`` Restrict the list of matches to the ones of a given
        minimal length. Can be provided only when ``idx`` set to True.
        ``withmatchlen`` Returns the matches with the len of the match.
        Can be provided only when ``idx`` set to True.

        For more information see https://redis.io/commands/stralgo
        """
    supported_algo = ['LCS']
    if algo not in supported_algo:
        supported_algos_str = ', '.join(supported_algo)
        raise DataError(f'The supported algorithms are: {supported_algos_str}')
    if specific_argument not in ['keys', 'strings']:
        raise DataError('specific_argument can be only keys or strings')
    if len and idx:
        raise DataError('len and idx cannot be provided together.')
    pieces: list[EncodableT] = [algo, specific_argument.upper(), value1, value2]
    if len:
        pieces.append(b'LEN')
    if idx:
        pieces.append(b'IDX')
    try:
        int(minmatchlen)
        pieces.extend([b'MINMATCHLEN', minmatchlen])
    except TypeError:
        pass
    if withmatchlen:
        pieces.append(b'WITHMATCHLEN')
    return self.execute_command('STRALGO', *pieces, len=len, idx=idx, minmatchlen=minmatchlen, withmatchlen=withmatchlen, **kwargs)