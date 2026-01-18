from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.experimental.enable_op_determinism', v1=[])
def enable_op_determinism():
    """Configures TensorFlow ops to run deterministically.

  When op determinism is enabled, TensorFlow ops will be deterministic. This
  means that if an op is run multiple times with the same inputs on the same
  hardware, it will have the exact same outputs each time. This is useful for
  debugging models. Note that determinism in general comes at the expense of
  lower performance and so your model may run slower when op determinism is
  enabled.

  If you want your TensorFlow program to run deterministically, put the
  following code near the start of your program.

  ```python
  tf.keras.utils.set_random_seed(1)
  tf.config.experimental.enable_op_determinism()
  ```

  Calling `tf.keras.utils.set_random_seed` sets the Python seed, the NumPy seed,
  and the TensorFlow seed. Setting these seeds is necessary to ensure any random
  numbers your program generates are also deterministic.

  By default, op determinism is not enabled, so ops might return different
  results when run with the same inputs. These differences are often caused by
  the use of asynchronous threads within the op nondeterministically changing
  the order in which floating-point numbers are added. Most of these cases of
  nondeterminism occur on GPUs, which have thousands of hardware threads that
  are used to run ops. Enabling determinism directs such ops to use a different
  algorithm, one that does not use threads in a nondeterministic way.

  Another potential source of nondeterminism is `tf.data` based data processing.
  Typically, this can introduce nondeterminsm due to the use of parallelism in
  methods such as `Dataset.map` producing inputs or running stateful ops in a
  nondeterministic order. Enabling determinism will remove such sources of
  nondeterminism.

  Enabling determinism will likely make your model or your `tf.data` data
  processing slower. For example, `Dataset.map` can become several orders of
  magnitude slower when the map function has random ops or other stateful ops.
  See the “Determinism and tf.data” section below for more details. In future
  TensorFlow releases, we plan on improving the performance of determinism,
  especially for common scenarios such as `Dataset.map`.

  Certain ops will raise an `UnimplementedError` because they do not yet have a
  deterministic implementation. Additionally, due to bugs, some ops might be
  nondeterministic and not raise an `UnimplementedError`. If you encounter such
  ops, please [file an issue](https://github.com/tensorflow/tensorflow/issues).

  An example of enabling determinism follows. The
  `tf.nn.softmax_cross_entropy_with_logits` op is run multiple times and the
  output is shown to be the same each time. This example would likely fail when
  run on a GPU if determinism were not enabled, because
  `tf.nn.softmax_cross_entropy_with_logits` uses a nondeterministic algorithm on
  GPUs by default.

  ```python
  labels = tf.random.normal((1, 10000))
  logits = tf.random.normal((1, 10000))
  output = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                   logits=logits)
  for _ in range(5):
    output2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                      logits=logits)
    tf.debugging.assert_equal(output, output2)
  ```

  ## Writing deterministic models

  You can make your models deterministic by enabling op determinism. This
  means that you can train a model and finish each run with exactly the same
  trainable variables. This also means that the inferences of your
  previously-trained model will be exactly the same on each run. Typically,
  models can be made deterministic by simply setting the seeds and enabling
  op determinism, as in the example above. However, to guarantee that your
  model operates deterministically, you must meet all the following
  requirements:

  * Call `tf.config.experimental.enable_op_determinism()`, as mentioned above.
  * Reproducibly reset any pseudorandom number generators (PRNGs) you’re using,
    such as by setting the seeds for the default PRNGs in TensorFlow, Python,
    and NumPy, as mentioned above. Note that certain newer NumPy classes like
   ` numpy.random.default_rng` ignore the global NumPy seed, so a seed must be
    explicitly passed to such classes, if used.
  * Use the same hardware configuration in every run.
  * Use the same software environment in every run (OS, checkpoints, version of
    CUDA and TensorFlow, environmental variables, etc). Note that determinism is
    not guaranteed across different versions of TensorFlow.
  * Do not use constructs outside TensorFlow that are nondeterministic, such as
    reading from `/dev/random` or using multiple threads/processes in ways that
    influence TensorFlow’s behavior.
  * Ensure your input pipeline is deterministic. If you use `tf.data`, this is
    done automatically (at the expense of performance). See "Determinism and
    tf.data" below for more information.
  * Do not use `tf.compat.v1.Session` and
    `tf.distribute.experimental.ParameterServerStrategy`, which can introduce
    nondeterminism. Besides ops (including `tf.data` ops), these are the only
    known potential sources of nondeterminism within TensorFlow, (if you
    find more, please file an issue). Note that `tf.compat.v1.Session` is
    required to use the TF1 API, so determinism cannot be guaranteed when using
    the TF1 API.
  * Do not use nondeterministic custom ops.

  ## Additional details on determinism

  For stateful ops to be deterministic, the state of the system must be the same
  every time the op is run. For example the output of `tf.Variable.sparse_read`
  (obviously) depends on both the variable value and the `indices` function
  parameter.  When determinism is enabled, the side effects of stateful ops are
  deterministic.

  TensorFlow’s random ops, such as `tf.random.normal`, will raise a
  `RuntimeError` if determinism is enabled and a seed has not been set. However,
  attempting to generate nondeterministic random numbers using Python or NumPy
  will not raise such errors. Make sure you remember to set the Python and NumPy
  seeds. Calling `tf.keras.utils.set_random_seed` is an easy way to set all
  three seeds.

  Note that latency, memory consumption, throughput, and other performance
  characteristics are *not* made deterministic by enabling op determinism.
  Only op outputs and side effects are made deterministic. Additionally, a model
  may nondeterministically raise a `tf.errors.ResourceExhaustedError` from a
  lack of memory due to the fact that memory consumption is nondeterministic.

  ## Determinism and tf.data

  Enabling deterministic ops makes `tf.data` deterministic in several ways:

  1. For dataset methods with a `deterministic` argument, such as `Dataset.map`
     and `Dataset.batch`, the `deterministic` argument is overridden to be
     `True` irrespective of its setting.
  2. The `tf.data.Option.experimental_deterministic` option is overridden to be
     `True` irrespective of its setting..
  3. In `Dataset.map` and `Dataset.interleave`, if the map or interleave
     function has stateful random ops or other stateful ops, the function will
     run serially instead of in parallel. This means the `num_parallel_calls`
     argument to `map` and `interleave` is effectively ignored.
  4. Prefetching with `Dataset.prefetch` will be disabled if any function run
     as part of the input pipeline has certain stateful ops. Similarly, any
     dataset method with a `num_parallel_calls` argument will be made to run
     serially if any function in the input pipeline has such stateful ops.
     Legacy random ops such as `tf.random.normal` will *not* cause such datasets
     to be changed, but most other stateful ops will.

  Unfortunately, due to (3), performance can be greatly reduced when stateful
  ops are used in `Dataset.map` due to no longer running the map function in
  parallel. A common example of stateful ops used in `Dataset.map` are random
  ops, such as `tf.random.normal`, which are typically used for distortions. One
  way to work around this is to use stateless random ops instead. Alternatively
  you can hoist all random ops into its own separate `Dataset.map` call, making
  the original `Dataset.map` call stateless and thus avoid the need to serialize
  its execution.

  (4) can also cause performance to be reduced, but occurs less frequently than
  (3) because legacy random ops do not cause (4) to take effect. However, unlike
  (3), when there are non-random stateful ops in a user-defined function, every
  `map` and `interleave` dataset is affected, instead of just the `map` or
  `interleave` dataset with the function that has stateful ops. Additionally,
  `prefetch` datasets and any dataset with the `num_parallel_calls` argument are
  also affected.
  """
    _pywrap_determinism.enable(True)