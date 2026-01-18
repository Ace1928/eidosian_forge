import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def sample_distorted_bounding_box(image_size: _atypes.TensorFuzzingAnnotation[TV_SampleDistortedBoundingBox_T], bounding_boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], seed: int=0, seed2: int=0, min_object_covered: float=0.1, aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1], max_attempts: int=100, use_image_if_no_bounding_boxes: bool=False, name=None):
    """Generate a single randomly distorted bounding box for an image.

  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.

  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
  what the bounding box looks like.

  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example,

  ```python
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes)

      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.summary.image('images_with_box', image_with_box)

      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)
  ```

  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.

  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`.
      1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, N, 4]` describing the N bounding boxes
      associated with the image.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to non-zero, the random number
      generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
      seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    min_object_covered: An optional `float`. Defaults to `0.1`.
      The cropped area of the image must contain at least this
      fraction of any bounding box supplied. The value of this parameter should be
      non-negative. In the case of 0, the cropped area does not need to overlap
      any of the bounding boxes supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75, 1.33]`.
      The cropped area of the image must have an aspect ratio =
      width / height within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`.
      The cropped area of the image must contain a fraction of the
      supplied image within this range.
    max_attempts: An optional `int`. Defaults to `100`.
      Number of attempts at generating a cropped region of the image
      of the specified constraints. After `max_attempts` failures, return the entire
      image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied.
      If true, assume an implicit bounding box covering the whole input. If false,
      raise an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).

    begin: A `Tensor`. Has the same type as `image_size`.
    size: A `Tensor`. Has the same type as `image_size`.
    bboxes: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SampleDistortedBoundingBox', name, image_size, bounding_boxes, 'seed', seed, 'seed2', seed2, 'min_object_covered', min_object_covered, 'aspect_ratio_range', aspect_ratio_range, 'area_range', area_range, 'max_attempts', max_attempts, 'use_image_if_no_bounding_boxes', use_image_if_no_bounding_boxes)
            _result = _SampleDistortedBoundingBoxOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sample_distorted_bounding_box_eager_fallback(image_size, bounding_boxes, seed=seed, seed2=seed2, min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    if min_object_covered is None:
        min_object_covered = 0.1
    min_object_covered = _execute.make_float(min_object_covered, 'min_object_covered')
    if aspect_ratio_range is None:
        aspect_ratio_range = [0.75, 1.33]
    if not isinstance(aspect_ratio_range, (list, tuple)):
        raise TypeError("Expected list for 'aspect_ratio_range' argument to 'sample_distorted_bounding_box' Op, not %r." % aspect_ratio_range)
    aspect_ratio_range = [_execute.make_float(_f, 'aspect_ratio_range') for _f in aspect_ratio_range]
    if area_range is None:
        area_range = [0.05, 1]
    if not isinstance(area_range, (list, tuple)):
        raise TypeError("Expected list for 'area_range' argument to 'sample_distorted_bounding_box' Op, not %r." % area_range)
    area_range = [_execute.make_float(_f, 'area_range') for _f in area_range]
    if max_attempts is None:
        max_attempts = 100
    max_attempts = _execute.make_int(max_attempts, 'max_attempts')
    if use_image_if_no_bounding_boxes is None:
        use_image_if_no_bounding_boxes = False
    use_image_if_no_bounding_boxes = _execute.make_bool(use_image_if_no_bounding_boxes, 'use_image_if_no_bounding_boxes')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SampleDistortedBoundingBox', image_size=image_size, bounding_boxes=bounding_boxes, seed=seed, seed2=seed2, min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'min_object_covered', _op.get_attr('min_object_covered'), 'aspect_ratio_range', _op.get_attr('aspect_ratio_range'), 'area_range', _op.get_attr('area_range'), 'max_attempts', _op._get_attr_int('max_attempts'), 'use_image_if_no_bounding_boxes', _op._get_attr_bool('use_image_if_no_bounding_boxes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SampleDistortedBoundingBox', _inputs_flat, _attrs, _result)
    _result = _SampleDistortedBoundingBoxOutput._make(_result)
    return _result