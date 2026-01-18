from tensorboard.compat import tf2 as tf
from tensorboard.plugins.image import metadata
from tensorboard.util import lazy_tensor_creator
@lazy_tensor_creator.LazyTensorCreator
def lazy_tensor():
    tf.debugging.assert_rank(data, 4)
    tf.debugging.assert_non_negative(max_outputs)
    images = tf.image.convert_image_dtype(data, tf.uint8, saturate=True)
    limited_images = images[:max_outputs]
    if tf.compat.forward_compatible(2023, 5, 1):
        encoded_images = tf.image.encode_png(limited_images)
    else:
        encoded_images = tf.map_fn(tf.image.encode_png, limited_images, dtype=tf.string, name='encode_each_image')
        encoded_images = tf.cond(tf.shape(input=encoded_images)[0] > 0, lambda: encoded_images, lambda: tf.constant([], tf.string))
    image_shape = tf.shape(input=images)
    dimensions = tf.stack([tf.as_string(image_shape[2], name='width'), tf.as_string(image_shape[1], name='height')], name='dimensions')
    return tf.concat([dimensions, encoded_images], axis=0)