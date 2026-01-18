import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
def optimize_for(self, x, *args, backend=None, clear=False, static_alloc=False, static_shape=False, inline_limit=2, forward_bulk_size=None, backward_bulk_size=None, **kwargs):
    """Partitions the current HybridBlock and optimizes it for a given backend
        without executing a forward pass. Modifies the HybridBlock in-place.

        Immediately partitions a HybridBlock using the specified backend. Combines
        the work done in the hybridize API with part of the work done in the forward
        pass without calling the CachedOp. Can be used in place of hybridize,
        afterwards `export` can be called or inference can be run. See README.md in
        example/extensions/lib_subgraph/README.md for more details.

        Examples
        --------
        # partition and then export to file
        block.optimize_for(x, backend='myPart')
        block.export('partitioned')

        # partition and then run inference
        block.optimize_for(x, backend='myPart')
        block(x)

        Parameters
        ----------
        x : NDArray
            first input to model
        *args : NDArray
            other inputs to model
        backend : str
            The name of backend, as registered in `SubgraphBackendRegistry`, default None
        clear : bool, default False
            Clears any previous optimizations
        static_alloc : bool, default False
            Statically allocate memory to improve speed. Memory usage may increase.
        static_shape : bool, default False
            Optimize for invariant input shapes between iterations. Must also
            set static_alloc to True. Change of input shapes is still allowed
            but slower.
        inline_limit : optional int, default 2
            Maximum number of operators that can be inlined.
        forward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        backward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        **kwargs: The backend options, optional
            Passed on to `PrePartition` and `PostPartition` functions of `SubgraphProperty`
        """
    if len(kwargs) > 0:
        self._backend_opts = kwargs
    if not backend:
        raise ValueError('Must specify "backend" to optimize_for')
    self.hybridize(True, backend, clear, static_alloc, static_shape, inline_limit, forward_bulk_size, backward_bulk_size)
    has_symbol, has_ndarray, ctx_set, _ = _gather_type_ctx_info([x] + list(args))
    if not has_symbol and (not has_ndarray):
        raise ValueError('In HybridBlock, there must be one NDArray or one Symbol in the input. Please check the type of the args.\n')
    if len(ctx_set) > 1:
        raise ValueError('Found multiple contexts in the input, After hybridized, the HybridBlock only supports one input context. You can print the ele.ctx in the input arguments to inspect their contexts. Find all contexts = {}'.format(ctx_set))
    self._build_cache(x, *args)
    assert self._cached_op, 'Gluon failed to build the cache. This should never happen. Please submit an issue on Github https://github.com/apache/incubator-mxnet.'