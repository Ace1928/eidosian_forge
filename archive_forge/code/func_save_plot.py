import os
def save_plot(fig, file_name=None):
    try:
        from plotly.offline import iplot, init_notebook_mode
        init_notebook_mode(connected=True)
        iplot(fig, filename=file_name)
    except Exception:
        from plotly.offline import plot
        plot(fig, auto_open=False, filename=file_name)