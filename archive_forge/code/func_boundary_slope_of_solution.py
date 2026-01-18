import snappy
import FXrays
def boundary_slope_of_solution(self, quad_vector):
    return self.slope_matrix * self.shift_matrix * quad_vector