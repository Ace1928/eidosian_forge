import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def get_streamline_arrows(self):
    """
        Makes an arrow for each streamline.

        Gets angle of streamline at 1/3 mark and creates arrow coordinates
        based off of user defined angle and arrow_scale.

        :param (array) st_x: x-values for all streamlines
        :param (array) st_y: y-values for all streamlines
        :param (angle in radians) angle: angle of arrowhead. Default = pi/9
        :param (float in [0,1]) arrow_scale: value to scale length of arrowhead
            Default = .09
        :rtype (list, list) arrows_x: x-values to create arrowhead and
            arrows_y: y-values to create arrowhead
        """
    arrow_end_x = np.empty(len(self.st_x))
    arrow_end_y = np.empty(len(self.st_y))
    arrow_start_x = np.empty(len(self.st_x))
    arrow_start_y = np.empty(len(self.st_y))
    for index in range(len(self.st_x)):
        arrow_end_x[index] = self.st_x[index][int(len(self.st_x[index]) / 3)]
        arrow_start_x[index] = self.st_x[index][int(len(self.st_x[index]) / 3) - 1]
        arrow_end_y[index] = self.st_y[index][int(len(self.st_y[index]) / 3)]
        arrow_start_y[index] = self.st_y[index][int(len(self.st_y[index]) / 3) - 1]
    dif_x = arrow_end_x - arrow_start_x
    dif_y = arrow_end_y - arrow_start_y
    orig_err = np.geterr()
    np.seterr(divide='ignore', invalid='ignore')
    streamline_ang = np.arctan(dif_y / dif_x)
    np.seterr(**orig_err)
    ang1 = streamline_ang + self.angle
    ang2 = streamline_ang - self.angle
    seg1_x = np.cos(ang1) * self.arrow_scale
    seg1_y = np.sin(ang1) * self.arrow_scale
    seg2_x = np.cos(ang2) * self.arrow_scale
    seg2_y = np.sin(ang2) * self.arrow_scale
    point1_x = np.empty(len(dif_x))
    point1_y = np.empty(len(dif_y))
    point2_x = np.empty(len(dif_x))
    point2_y = np.empty(len(dif_y))
    for index in range(len(dif_x)):
        if dif_x[index] >= 0:
            point1_x[index] = arrow_end_x[index] - seg1_x[index]
            point1_y[index] = arrow_end_y[index] - seg1_y[index]
            point2_x[index] = arrow_end_x[index] - seg2_x[index]
            point2_y[index] = arrow_end_y[index] - seg2_y[index]
        else:
            point1_x[index] = arrow_end_x[index] + seg1_x[index]
            point1_y[index] = arrow_end_y[index] + seg1_y[index]
            point2_x[index] = arrow_end_x[index] + seg2_x[index]
            point2_y[index] = arrow_end_y[index] + seg2_y[index]
    space = np.empty(len(point1_x))
    space[:] = np.nan
    arrows_x = np.array([point1_x, arrow_end_x, point2_x, space])
    arrows_x = arrows_x.flatten('F')
    arrows_x = arrows_x.tolist()
    arrows_y = np.array([point1_y, arrow_end_y, point2_y, space])
    arrows_y = arrows_y.flatten('F')
    arrows_y = arrows_y.tolist()
    return (arrows_x, arrows_y)