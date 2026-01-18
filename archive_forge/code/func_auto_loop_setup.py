def auto_loop_setup(use_subprocess: bool=False) -> None:
    try:
        import uvloop
    except ImportError:
        from uvicorn.loops.asyncio import asyncio_setup as loop_setup
        loop_setup(use_subprocess=use_subprocess)
    else:
        from uvicorn.loops.uvloop import uvloop_setup
        uvloop_setup(use_subprocess=use_subprocess)