import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo
def make_sprite(label_img, save_path):
    from PIL import Image
    from io import BytesIO
    nrow = int(math.ceil(label_img.size(0) ** 0.5))
    arranged_img_CHW = make_grid(make_np(label_img), ncols=nrow)
    arranged_augment_square_HWC = np.zeros((arranged_img_CHW.shape[2], arranged_img_CHW.shape[2], 3))
    arranged_img_HWC = arranged_img_CHW.transpose(1, 2, 0)
    arranged_augment_square_HWC[:arranged_img_HWC.shape[0], :, :] = arranged_img_HWC
    im = Image.fromarray(np.uint8((arranged_augment_square_HWC * 255).clip(0, 255)))
    with BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    with tf.io.gfile.GFile(_gfile_join(save_path, 'sprite.png'), 'wb') as f:
        f.write(im_bytes)