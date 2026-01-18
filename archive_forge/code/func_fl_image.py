def fl_image(im):
    im = 1.0 * im
    corrected = im + lum + contrast * (im - float(contrast_thr))
    corrected[corrected < 0] = 0
    corrected[corrected > 255] = 255
    return corrected.astype('uint8')