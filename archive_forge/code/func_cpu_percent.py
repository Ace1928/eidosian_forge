import logging
from ray._private.utils import get_num_cpus
def cpu_percent():
    """Estimate CPU usage percent for Ray pod managed by Kubernetes
    Operator.

    Computed by the following steps
       (1) Replicate the logic used by 'docker stats' cli command.
           See https://github.com/docker/cli/blob/c0a6b1c7b30203fbc28cd619acb901a95a80e30e/cli/command/container/stats_helpers.go#L166.
       (2) Divide by the number of CPUs available to the container, so that
           e.g. full capacity use of 2 CPUs will read as 100%,
           rather than 200%.

    Step (1) above works by
        dividing delta in cpu usage by
        delta in total host cpu usage, averaged over host's cpus.

    Since deltas are not initially available, return 0.0 on first call.
    """
    global last_system_usage
    global last_cpu_usage
    try:
        cpu_usage = _cpu_usage()
        system_usage = _system_usage()
        if last_system_usage is None:
            cpu_percent = 0.0
        else:
            cpu_delta = cpu_usage - last_cpu_usage
            system_delta = (system_usage - last_system_usage) / _host_num_cpus()
            quotient = cpu_delta / system_delta
            cpu_percent = round(quotient * 100 / get_num_cpus(), 1)
        last_system_usage = system_usage
        last_cpu_usage = cpu_usage
        return min(cpu_percent, 100.0)
    except Exception:
        logger.exception('Error computing CPU usage of Ray Kubernetes pod.')
        return 0.0