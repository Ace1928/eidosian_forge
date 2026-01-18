import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data, run_cstr_mpc
def test_mpc_simulation(self):
    initial_data = self._get_initial_data()
    setpoint_data = self._get_setpoint_data()
    sample_time = 2.0
    samples_per_horizon = 5
    ntfe_per_sample = 2
    ntfe_plant = 5
    simulation_steps = 5
    m_plant, sim_data = run_cstr_mpc(initial_data, setpoint_data, samples_per_controller_horizon=samples_per_horizon, sample_time=sample_time, ntfe_per_sample_controller=ntfe_per_sample, ntfe_plant=ntfe_plant, simulation_steps=simulation_steps)
    sim_time_points = [sample_time / ntfe_plant * i for i in range(simulation_steps * ntfe_plant + 1)]
    AB_data = sim_data.extract_variables([m_plant.conc[:, 'A'], m_plant.conc[:, 'B']])
    A_cuid = sim_data.get_cuid(m_plant.conc[:, 'A'])
    B_cuid = sim_data.get_cuid(m_plant.conc[:, 'B'])
    pred_data = {A_cuid: self._pred_A_data, B_cuid: self._pred_B_data}
    self.assertStructuredAlmostEqual(pred_data, AB_data.get_data(), delta=0.001)
    self.assertStructuredAlmostEqual(sim_time_points, AB_data.get_time_points(), delta=1e-07)