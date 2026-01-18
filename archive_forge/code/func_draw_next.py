import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def draw_next(self, color=None, thickness=2, mean=None, std=None, clip=True, waitKey=None, window_name='draw_next', id2labels=None):
    """Display next image with bounding boxes drawn.

        Parameters
        ----------
        color : tuple
            Bounding box color in RGB, use None for random color
        thickness : int
            Bounding box border thickness
        mean : True or numpy.ndarray
            Compensate for the mean to have better visual effect
        std : True or numpy.ndarray
            Revert standard deviations
        clip : bool
            If true, clip to [0, 255] for better visual effect
        waitKey : None or int
            Hold the window for waitKey milliseconds if set, skip ploting if None
        window_name : str
            Plot window name if waitKey is set.
        id2labels : dict
            Mapping of labels id to labels name.

        Returns
        -------
            numpy.ndarray

        Examples
        --------
        >>> # use draw_next to get images with bounding boxes drawn
        >>> iterator = mx.image.ImageDetIter(1, (3, 600, 600), path_imgrec='train.rec')
        >>> for image in iterator.draw_next(waitKey=None):
        ...     # display image
        >>> # or let draw_next display using cv2 module
        >>> for image in iterator.draw_next(waitKey=0, window_name='disp'):
        ...     pass
        """
    try:
        import cv2
    except ImportError as e:
        warnings.warn('Unable to import cv2, skip drawing: %s', str(e))
        return
    count = 0
    try:
        while True:
            label, s = self.next_sample()
            data = self.imdecode(s)
            try:
                self.check_valid_image([data])
                label = self._parse_label(label)
            except RuntimeError as e:
                logging.debug('Invalid image, skipping:  %s', str(e))
                continue
            count += 1
            data, label = self.augmentation_transform(data, label)
            image = data.asnumpy()
            if std is True:
                std = np.array([58.395, 57.12, 57.375])
            elif std is not None:
                assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]
            if std is not None:
                image *= std
            if mean is True:
                mean = np.array([123.68, 116.28, 103.53])
            elif mean is not None:
                assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]
            if mean is not None:
                image += mean
            image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]
            if clip:
                image = np.maximum(0, np.minimum(255, image))
            if color:
                color = color[::-1]
            image = image.astype(np.uint8)
            height, width, _ = image.shape
            for i in range(label.shape[0]):
                x1 = int(label[i, 1] * width)
                if x1 < 0:
                    continue
                y1 = int(label[i, 2] * height)
                x2 = int(label[i, 3] * width)
                y2 = int(label[i, 4] * height)
                bc = np.random.rand(3) * 255 if not color else color
                cv2.rectangle(image, (x1, y1), (x2, y2), bc, thickness)
                if id2labels is not None:
                    cls_id = int(label[i, 0])
                    if cls_id in id2labels:
                        cls_name = id2labels[cls_id]
                        text = '{:s}'.format(cls_name)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        text_height = cv2.getTextSize(text, font, font_scale, 2)[0][1]
                        tc = (255, 255, 255)
                        tpos = (x1 + 5, y1 + text_height + 5)
                        cv2.putText(image, text, tpos, font, font_scale, tc, 2)
            if waitKey is not None:
                cv2.imshow(window_name, image)
                cv2.waitKey(waitKey)
            yield image
    except StopIteration:
        if not count:
            return