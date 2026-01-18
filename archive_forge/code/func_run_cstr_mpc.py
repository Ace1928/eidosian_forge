import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.model import (
def run_cstr_mpc(initial_data, setpoint_data, samples_per_controller_horizon=5, sample_time=2.0, ntfe_per_sample_controller=2, ntfe_plant=5, simulation_steps=5, tee=False):
    controller_horizon = sample_time * samples_per_controller_horizon
    ntfe = ntfe_per_sample_controller * samples_per_controller_horizon
    m_controller = create_instance(horizon=controller_horizon, ntfe=ntfe)
    controller_interface = mpc.DynamicModelInterface(m_controller, m_controller.time)
    t0_controller = m_controller.time.first()
    m_plant = create_instance(horizon=sample_time, ntfe=ntfe_plant)
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.time)
    controller_interface.load_data(initial_data)
    plant_interface.load_data(initial_data)
    setpoint_variables = [m_controller.conc[:, 'A'], m_controller.conc[:, 'B']]
    vset, tr_cost = controller_interface.get_penalty_from_target(setpoint_data, variables=setpoint_variables)
    m_controller.setpoint_set = vset
    m_controller.tracking_cost = tr_cost
    m_controller.objective = pyo.Objective(expr=sum((m_controller.tracking_cost[i, t] for i in m_controller.setpoint_set for t in m_controller.time if t != m_controller.time.first())))
    m_controller.flow_in[:].unfix()
    m_controller.flow_in[t0_controller].fix()
    sample_points = [i * sample_time for i in range(samples_per_controller_horizon + 1)]
    input_set, pwc_con = controller_interface.get_piecewise_constant_constraints([m_controller.flow_in], sample_points)
    m_controller.input_set = input_set
    m_controller.pwc_con = pwc_con
    sim_t0 = 0.0
    sim_data = plant_interface.get_data_at_time([sim_t0])
    solver = pyo.SolverFactory('ipopt')
    non_initial_plant_time = list(m_plant.time)[1:]
    ts = sample_time + t0_controller
    for i in range(simulation_steps):
        sim_t0 = i * sample_time
        res = solver.solve(m_controller, tee=tee)
        pyo.assert_optimal_termination(res)
        ts_data = controller_interface.get_data_at_time(ts)
        input_data = ts_data.extract_variables([m_controller.flow_in])
        plant_interface.load_data(input_data)
        res = solver.solve(m_plant, tee=tee)
        pyo.assert_optimal_termination(res)
        m_data = plant_interface.get_data_at_time(non_initial_plant_time)
        m_data.shift_time_points(sim_t0 - m_plant.time.first())
        sim_data.concatenate(m_data)
        tf_data = plant_interface.get_data_at_time(m_plant.time.last())
        plant_interface.load_data(tf_data)
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(tf_data, time_points=t0_controller)
    return (m_plant, sim_data)