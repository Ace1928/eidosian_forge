def read_caffe_mean(caffe_mean_file):
    """
    Reads caffe formatted mean file
    :param caffe_mean_file: path to caffe mean file, presumably with 'binaryproto' suffix
    :return: mean image, converted from BGR to RGB format
    """
    import caffe_parser
    import numpy as np
    mean_blob = caffe_parser.caffe_pb2.BlobProto()
    with open(caffe_mean_file, 'rb') as f:
        mean_blob.ParseFromString(f.read())
    img_mean_np = np.array(mean_blob.data)
    img_mean_np = img_mean_np.reshape(mean_blob.channels, mean_blob.height, mean_blob.width)
    img_mean_np[[0, 2], :, :] = img_mean_np[[2, 0], :, :]
    return img_mean_np