from ._internal import NDArrayBase
from ..base import _Null
def box_nms(data=None, overlap_thresh=_Null, valid_thresh=_Null, topk=_Null, coord_start=_Null, score_index=_Null, id_index=_Null, background_id=_Null, force_suppress=_Null, in_format=_Null, out_format=_Null, out=None, name=None, **kwargs):
    """Apply non-maximum suppression to input.

    The output will be sorted in descending order according to `score`. Boxes with
    overlaps larger than `overlap_thresh`, smaller scores and background boxes
    will be removed and filled with -1, the corresponding position will be recorded
    for backward propogation.

    During back-propagation, the gradient will be copied to the original
    position according to the input index. For positions that have been suppressed,
    the in_grad will be assigned 0.
    In summary, gradients are sticked to its boxes, will either be moved or discarded
    according to its original index in input.

    Input requirements::

      1. Input tensor have at least 2 dimensions, (n, k), any higher dims will be regarded
      as batch, e.g. (a, b, c, d, n, k) == (a*b*c*d, n, k)
      2. n is the number of boxes in each batch
      3. k is the width of each box item.

    By default, a box is [id, score, xmin, ymin, xmax, ymax, ...],
    additional elements are allowed.

    - `id_index`: optional, use -1 to ignore, useful if `force_suppress=False`, which means
      we will skip highly overlapped boxes if one is `apple` while the other is `car`.

    - `background_id`: optional, default=-1, class id for background boxes, useful
      when `id_index >= 0` which means boxes with background id will be filtered before nms.

    - `coord_start`: required, default=2, the starting index of the 4 coordinates.
      Two formats are supported:

        - `corner`: [xmin, ymin, xmax, ymax]

        - `center`: [x, y, width, height]

    - `score_index`: required, default=1, box score/confidence.
      When two boxes overlap IOU > `overlap_thresh`, the one with smaller score will be suppressed.

    - `in_format` and `out_format`: default='corner', specify in/out box formats.

    Examples::

      x = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [1, 0.4, 0.1, 0.1, 0.2, 0.2],
           [0, 0.3, 0.1, 0.1, 0.14, 0.14], [2, 0.6, 0.5, 0.5, 0.7, 0.8]]
      box_nms(x, overlap_thresh=0.1, coord_start=2, score_index=1, id_index=0,
          force_suppress=True, in_format='corner', out_typ='corner') =
          [[2, 0.6, 0.5, 0.5, 0.7, 0.8], [0, 0.5, 0.1, 0.1, 0.2, 0.2],
           [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
      out_grad = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]
      # exe.backward
      in_grad = [[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]



    Defined in ../src/operator/contrib/bounding_box.cc:L94

    Parameters
    ----------
    data : NDArray
        The input
    overlap_thresh : float, optional, default=0.5
        Overlapping(IoU) threshold to suppress object with smaller score.
    valid_thresh : float, optional, default=0
        Filter input boxes to those whose scores greater than valid_thresh.
    topk : int, optional, default='-1'
        Apply nms to topk boxes with descending scores, -1 to no restriction.
    coord_start : int, optional, default='2'
        Start index of the consecutive 4 coordinates.
    score_index : int, optional, default='1'
        Index of the scores/confidence of boxes.
    id_index : int, optional, default='-1'
        Optional, index of the class categories, -1 to disable.
    background_id : int, optional, default='-1'
        Optional, id of the background class which will be ignored in nms.
    force_suppress : boolean, optional, default=0
        Optional, if set false and id_index is provided, nms will only apply to boxes belongs to the same category
    in_format : {'center', 'corner'},optional, default='corner'
        The input box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].
    out_format : {'center', 'corner'},optional, default='corner'
        The output box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)