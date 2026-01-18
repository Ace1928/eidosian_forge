import json
def get_iframe_html(run):
    return f'<iframe src="{run.url}?kfp=true" style="border:none;width:100%;height:100%;min-width:900px;min-height:600px;"></iframe>'